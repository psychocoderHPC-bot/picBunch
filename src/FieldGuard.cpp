// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: Rene Widera

#include "picBunch/detail/FieldGuard.hpp"

#include <sstream>
#include <stdexcept>

namespace picBunch::detail
{
    double totalCharge(Bunch const& bunch)
    {
        double sum = 0.0;
        for(std::size_t p = 0; p < bunch.size(); ++p)
            sum += bunch.weight[p] * bunch.charge[p];
        return sum;
    }

    bool fieldIsIdenticallyZero(Field const& field)
    {
        for(auto const& component : field.data)
            for(double const value : component)
                if(value != 0.0)
                    return false;
        return true;
    }

    namespace
    {
        [[noreturn]] void reportEmptyField(double charge, std::string const& configuration)
        {
            std::ostringstream message;
            message
                << "computeField: E and B are identically zero although the bunch carries a "
                   "non-zero total charge ("
                << std::scientific << charge
                << " C), so the source projection was skipped or wrote no charge. "
                   "Configuration: "
                << configuration << '.';
            throw std::runtime_error(message.str());
        }
    } // namespace

    void abortOnEmptyChargedField(
        Field const& field,
        double charge,
        std::string const& configuration)
    {
        if(charge == 0.0)
            return; // a neutral bunch has no field; this zero is physical
        if(!fieldIsIdenticallyZero(field))
            return;
        reportEmptyField(charge, configuration);
    }

    void abortOnEmptyChargedDistributedField(
        Field const& field,
        double charge,
        std::string const& configuration,
        ParallelContext const& par)
    {
        if(charge == 0.0)
            return;
        double localHasField = fieldIsIdenticallyZero(field) ? 0.0 : 1.0;
        par.allreduceSum(&localHasField, 1u);
        if(localHasField != 0.0)
            return;
        reportEmptyField(charge, configuration);
    }
} // namespace picBunch::detail
